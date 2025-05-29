SELECT 
	domain.name 
FROM 
	author, domain_author, domain
WHERE
	author.aid = domain_author.aid
    AND domain_author.did = domain.did
    AND author.name = 'Alin Deutsch'
GROUP BY domain.name 