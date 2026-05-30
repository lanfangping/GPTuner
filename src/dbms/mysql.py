from dbms.dbms_template import DBMSTemplate
import mysql.connector
import os
import json
import time
import subprocess

class MysqlDBMS(DBMSTemplate):
    """ Instantiate DBMSTemplate to support MySQL DBMS """
    def __init__(self, db, user, password, restart_cmd, recover_script, knob_info_path, host="localhost", port=3306):
        super().__init__(db, user, password, restart_cmd, recover_script, knob_info_path, host=host, port=port)
        self.name = "mysql"
        self.global_vars = [t[0] for t in (
            self.query_all('show global variables') or []) if self.is_numerical(t[1])]
        self.server_cost_params = [t[0] for t in (
            self.query_all('select cost_name from mysql.server_cost') or [])]
        self.engine_cost_params = [t[0] for t in (
            self.query_all('select cost_name from mysql.engine_cost') or [])]
        self.all_variables = self.global_vars + \
            self.server_cost_params + self.engine_cost_params
    
    def _connect(self, db=None):
        self.failed_times = 0
        if db==None:
            db=self.db
        print(f'Trying to connect to {db} with user {self.user}')
        while True:
            try:
                self.connection = mysql.connector.connect(
                    database=db,
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=int(self.port or 3306)
                )
                print(f"Success to connect to {db} with user {self.user}")
                return True
            except Exception as e:
                self.failed_times += 1
                print(f'Exception while trying to connect: {e}')
                if self.failed_times <= 4:
                    self.recover_dbms()
                    print("Reconnet again")
                else:
                    return False
                time.sleep(3)

            
    def _disconnect(self):
        if self.connection:
            print('Disconnecting ...')
            self.connection.close()
            print('Disconnecting done ...')
            self.connection = None
    
    def copy_db(self, source_db, target_db):
        container_name = os.environ.get("MYSQL_DOCKER_CONTAINER", "mysql-8")
        dump_path = "copy_db_dump"
        mysql_prefix = [
            "docker",
            "exec",
            container_name,
            "mysql",
            f"-u{self.user}",
            f"-p{self.password}",
        ]

        with open(dump_path, "wb") as dump_file:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "mysqldump",
                    f"-u{self.user}",
                    f"-p{self.password}",
                    "--single-transaction",
                    source_db,
                ],
                stdout=dump_file,
                check=True,
            )
        print('Dumped old database')
        subprocess.run(mysql_prefix + ["-e", f"drop database if exists `{target_db}`"], check=True)
        print('Dropped old database')
        subprocess.run(mysql_prefix + ["-e", f"create database `{target_db}`"], check=True)
        print('Created new database')
        with open(dump_path, "rb") as dump_file:
            subprocess.run(mysql_prefix + [target_db], stdin=dump_file, check=True)
        print('Initialized new database')

    def query_one(self, sql):
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            return cursor.fetchone()[0]
        except Exception:
            return None
    
    def query_all(self, sql):
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f'Exception in mysql.query_all: {e}')
            return None
    
    def reset_config(self):
        """ Reset all parameters to default values. """
        self._disconnect()
        self.recover_dbms()
        res = False
        while not res:
            print("Reconnecting for reconfiguring...")
            res = self._connect()
        self.update_dbms('update mysql.server_cost set cost_value = NULL')
        self.update_dbms('update mysql.engine_cost set cost_value = NULL')
        self.config = {}

    def reconfigure(self):
        """Makes all parameter changes take effect"""
        self.update_dbms('flush optimizer_costs')
        self._disconnect()
        success = self._connect()
        if success:
            return success
        else:
            try:
                self.recover_dbms()
                time.sleep(3)
                return True
            except Exception as e:
                print(f'Exception while trying to recover dbms: {e}')
                return False

    def update_dbms(self, sql):
        """ Execute sql query on dbms to update knob value and return success flag """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql, multi=True)
            cursor.close()
            return True
        except Exception as e:
            print(f"Failed to execute {sql} to update dbms for error: {e}")
            return False 

    def extract_knob_info(self, dest_path):
        """ Extract knob information and store the query result in json format """
        knob_info = {}
        knobs_sql = "SHOW VARIABLES;"
        knobs, _ = self.get_sql_result(knobs_sql)
        for knob in knobs:
            knob = knob[0] 
            knob_details_sql = f"SHOW VARIABLES WHERE VARIABLE_NAME = '{knob}';"
            knob_detail, description = self.get_sql_result(knob_details_sql)
            # print(knob, knob_detail)
            if knob_detail:
                column_names = [desc[0] for desc in description]
                knob_detail = knob_detail[0]
                knob_attributes = {}
                for i, column_name in enumerate(column_names):
                    knob_attributes[column_name] = knob_detail[i]
                knob_info[knob] = knob_attributes
            print(f"There are {len(knob_info)} knobs extracted.")
        with open(dest_path, 'w') as json_file:
            json.dump(knob_info, json_file, indent=4, sort_keys=True, default=self.datetime_serializer)
        print(f"The knob info is written to {dest_path}.")

    def get_sql_result(self, sql):
        """ Execute sql query on dbms and return the result and its description """
        self.connection.autocommit = True
        cursor = self.connection.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        description = cursor.description
        cursor.close()
        return result, description

    def set_knob(self, knob, knob_value):
        def mysql_literal(value):
            if isinstance(value, bool):
                return "ON" if value else "OFF"
            if isinstance(value, (int, float)):
                return str(value)

            value = str(value)
            try:
                float(value)
                return value
            except ValueError:
                escaped = value.replace("\\", "\\\\").replace("'", "\\'")
                return f"'{escaped}'"

        if knob in self.global_vars:
            success = self.update_dbms(f"set global `{knob}` = {mysql_literal(knob_value)}")
        elif knob in self.server_cost_params:
            success = self.update_dbms(
                f"update mysql.server_cost set cost_value={knob_value} where cost_name='{knob}'")
        elif knob in self.engine_cost_params:
            success = self.update_dbms(
                f"update mysql.engine_cost set cost_value={knob_value} where cost_name='{knob}'")
        else: 
            success = False
        if success:
            self.config[knob] = knob_value
        return success

    def get_knob_value(self, knob):
        cursor = self.connection.cursor()
        cursor.execute(f'SHOW VARIABLES LIKE "{knob}"')
        result = cursor.fetchone()
        cursor.close()
        return result

    def check_knob_exists(self, knob):
        cursor = self.connection.cursor()
        cursor.execute(f"SHOW VARIABLES LIKE '{knob}'")
        row = cursor.fetchone()
        cursor.close()
        return row is not None

    def exec_quries(self, sql):
        """ Executes all SQL queries in given file and returns success flag. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            sql_statements = sql.split(';')
            for statement in sql_statements:
                if statement.strip():
                    cursor.execute(statement)
            cursor.close()
            return True
        except Exception as e:
            print(f'Exception execution {sql}: {e}')
        return False
